from __future__ import print_function
import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six
from heapq import merge
from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
from .utilities import (key_name_to_key_number, qpm_to_bpm)
def adjust_times(self, original_times, new_times):
    """Adjusts the timing of the events in the MIDI object.
        The parameters ``original_times`` and ``new_times`` define a mapping,
        so that if an event originally occurs at time ``original_times[n]``, it
        will be moved so that it occurs at ``new_times[n]``.  If events don't
        occur exactly on a time in ``original_times``, their timing will be
        linearly interpolated.

        Parameters
        ----------
        original_times : np.ndarray
            Times to map from.
        new_times : np.ndarray
            New times to map to.

        """
    original_downbeats = self.get_downbeats()
    original_size = len(original_times)
    original_times, unique_idx = np.unique(original_times, return_index=True)
    if unique_idx.size != original_size or any(unique_idx != np.arange(unique_idx.size)):
        warnings.warn('original_times must be strictly increasing; automatically enforcing this.')
    new_times = np.asarray(new_times)[unique_idx]
    if not np.all(np.diff(new_times) >= 0):
        warnings.warn('new_times must be monotonic; automatically enforcing this.')
        new_times = np.maximum.accumulate(new_times)
    for instrument in self.instruments:
        instrument.notes = [copy.deepcopy(note) for note in instrument.notes if note.start >= original_times[0] and note.end <= original_times[-1]]
    note_ons = np.array([note.start for instrument in self.instruments for note in instrument.notes])
    adjusted_note_ons = np.interp(note_ons, original_times, new_times)
    note_offs = np.array([note.end for instrument in self.instruments for note in instrument.notes])
    adjusted_note_offs = np.interp(note_offs, original_times, new_times)
    for n, note in enumerate([note for instrument in self.instruments for note in instrument.notes]):
        note.start = (adjusted_note_ons[n] > 0) * adjusted_note_ons[n]
        note.end = (adjusted_note_offs[n] > 0) * adjusted_note_offs[n]
    self.remove_invalid_notes()

    def adjust_events(event_getter):
        """ This function calls event_getter with each instrument as the
            sole argument and adjusts the events which are returned."""
        for instrument in self.instruments:
            event_getter(instrument).sort(key=lambda e: e.time)
        event_times = np.array([event.time for instrument in self.instruments for event in event_getter(instrument)])
        adjusted_event_times = np.interp(event_times, original_times, new_times)
        for n, event in enumerate([event for instrument in self.instruments for event in event_getter(instrument)]):
            event.time = adjusted_event_times[n]
        for instrument in self.instruments:
            valid_events = [event for event in event_getter(instrument) if event.time == new_times[0]]
            if valid_events:
                valid_events = valid_events[-1:]
            valid_events.extend((event for event in event_getter(instrument) if event.time > new_times[0] and event.time < new_times[-1]))
            event_getter(instrument)[:] = valid_events
    adjust_events(lambda i: i.pitch_bends)
    adjust_events(lambda i: i.control_changes)

    def adjust_meta(events):
        """ This function adjusts the timing of the track-level meta-events
            in the provided list"""
        events.sort(key=lambda e: e.time)
        event_times = np.array([event.time for event in events])
        adjusted_event_times = np.interp(event_times, original_times, new_times)
        for event, adjusted_event_time in zip(events, adjusted_event_times):
            event.time = adjusted_event_time
        valid_events = [event for event in events if event.time == new_times[0]]
        if valid_events:
            valid_events = valid_events[-1:]
        valid_events.extend((event for event in events if event.time > new_times[0] and event.time < new_times[-1]))
        events[:] = valid_events
    adjust_meta(self.key_signature_changes)
    adjust_meta(self.lyrics)
    adjust_meta(self.text_events)
    original_downbeats = original_downbeats[original_downbeats >= original_times[0]]
    adjusted_downbeats = np.interp(original_downbeats, original_times, new_times)
    adjust_meta(self.time_signature_changes)
    if adjusted_downbeats.size > 0:
        ts_changes_before_downbeat = [t for t in self.time_signature_changes if t.time <= adjusted_downbeats[0]]
        if ts_changes_before_downbeat:
            ts_changes_before_downbeat[-1].time = adjusted_downbeats[0]
            self.time_signature_changes = [t for t in self.time_signature_changes if t.time >= adjusted_downbeats[0]]
        else:
            self.time_signature_changes.insert(0, TimeSignature(4, 4, adjusted_downbeats[0]))
    self._update_tick_to_time(self.time_to_tick(original_times[-1]))
    original_times = [self.__tick_to_time[self.time_to_tick(time)] for time in original_times]
    tempo_change_times, tempo_changes = self.get_tempo_changes()
    non_repeats = [0] + [n for n in range(1, len(new_times)) if new_times[n - 1] != new_times[n] and original_times[n - 1] != original_times[n]]
    new_times = [new_times[n] for n in non_repeats]
    original_times = [original_times[n] for n in non_repeats]
    speed_scales = np.diff(original_times) / np.diff(new_times)
    tempo_idx = 0
    while tempo_idx + 1 < len(tempo_changes) and original_times[0] >= tempo_change_times[tempo_idx + 1]:
        tempo_idx += 1
    new_tempo_change_times, new_tempo_changes = ([], [])
    for start_time, end_time, speed_scale in zip(original_times[:-1], original_times[1:], speed_scales):
        new_tempo_change_times.append(start_time)
        new_tempo_changes.append(tempo_changes[tempo_idx] * speed_scale)
        while tempo_idx + 1 < len(tempo_changes) and start_time <= tempo_change_times[tempo_idx + 1] and (end_time > tempo_change_times[tempo_idx + 1]):
            tempo_idx += 1
            new_tempo_change_times.append(tempo_change_times[tempo_idx])
            new_tempo_changes.append(tempo_changes[tempo_idx] * speed_scale)
    new_tempo_change_times = np.interp(new_tempo_change_times, original_times, new_times)
    if new_tempo_change_times[0] == 0:
        last_tick = 0
        new_tempo_change_times = new_tempo_change_times[1:]
        last_tick_scale = 60.0 / (new_tempo_changes[0] * self.resolution)
        new_tempo_changes = new_tempo_changes[1:]
    else:
        last_tick, last_tick_scale = (0, 60.0 / (120.0 * self.resolution))
    self._tick_scales = [(last_tick, last_tick_scale)]
    previous_time = 0.0
    for time, tempo in zip(new_tempo_change_times, new_tempo_changes):
        tick = last_tick + (time - previous_time) / last_tick_scale
        tick_scale = 60.0 / (tempo * self.resolution)
        if tick_scale != last_tick_scale:
            self._tick_scales.append((int(round(tick)), tick_scale))
            previous_time = time
            last_tick, last_tick_scale = (tick, tick_scale)
    self._update_tick_to_time(self._tick_scales[-1][0] + 1)