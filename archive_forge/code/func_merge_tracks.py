from .meta import MetaMessage
def merge_tracks(tracks, skip_checks=False):
    """Returns a MidiTrack object with all messages from all tracks.

    The messages are returned in playback order with delta times
    as if they were all in one track.

    Pass skip_checks=True to skip validation of messages before merging.
    This should ONLY be used when the messages in tracks have already
    been validated by mido.checks.
    """
    messages = []
    for track in tracks:
        messages.extend(_to_abstime(track, skip_checks=skip_checks))
    messages.sort(key=lambda msg: msg.time)
    return MidiTrack(fix_end_of_track(_to_reltime(messages, skip_checks=skip_checks), skip_checks=skip_checks))