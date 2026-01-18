from __future__ import absolute_import, division, print_function
import numpy as np
import mido
@classmethod
def from_notes(cls, notes, unit='seconds', tempo=DEFAULT_TEMPO, time_signature=DEFAULT_TIME_SIGNATURE, ticks_per_beat=DEFAULT_TICKS_PER_BEAT):
    """
        Create a MIDIFile from the given notes.

        Parameters
        ----------
        notes : numpy array
            Array with notes, one per row. The columns are defined as:
            (onset time, pitch, duration, velocity, [channel]).
        unit : str, optional
            Unit of `notes`, can be one of the following:

            - 'seconds', 's': use seconds as unit,
            - 'ticks', 't': use native MIDI ticks as unit,
            - 'beats', 'b' : use beats as unit.

        tempo : float, optional
            Tempo of the MIDI track, given in bpm or microseconds per quarter
            note. The unit is determined automatically by the value:

            - `tempo` <= 1000: bpm
            - `tempo` > 1000: microseconds per quarter note

        time_signature : tuple, optional
            Time signature of the track, e.g. (4, 4) for 4/4.
        ticks_per_beat : int, optional
            Resolution (i.e. ticks per quarter note) of the MIDI file.

        Returns
        -------
        :class:`MIDIFile` instance
            :class:`MIDIFile` instance with all notes collected in one track.

        Notes
        -----
        All note events (including the generated tempo and time signature
        events) are written into a single track (i.e. MIDI file format 0).

        """
    midi_file = cls(file_format=0, ticks_per_beat=ticks_per_beat, unit=unit, timing='absolute')
    if tempo <= 1000:
        tempo = bpm2tempo(tempo, time_signature)
    else:
        tempo = int(tempo * time_signature[1] / 4)
    track = midi_file.add_track()
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    track.append(mido.MetaMessage('time_signature', numerator=time_signature[0], denominator=time_signature[1]))
    messages = []
    for note in notes:
        try:
            onset, pitch, duration, velocity, channel = note
            channel = int(channel)
            velocity = int(velocity)
        except ValueError:
            onset, pitch, duration, velocity = note
            channel = 0
        pitch = int(pitch)
        velocity = int(velocity)
        offset = onset + duration
        onset = second2tick(onset, ticks_per_beat, tempo)
        note_on = mido.Message('note_on', time=onset, note=pitch, velocity=velocity, channel=channel)
        offset = second2tick(offset, ticks_per_beat, tempo)
        note_off = mido.Message('note_off', time=offset, note=pitch, channel=channel)
        messages.extend([note_on, note_off])
    messages.sort(key=lambda msg: msg.time)
    messages = mido.midifiles.tracks._to_reltime(messages)
    track.extend(messages)
    return midi_file