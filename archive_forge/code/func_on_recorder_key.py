from kivy.logger import Logger
from functools import partial
def on_recorder_key(recorder, window, key, *largs):
    if key == 289:
        if recorder.play:
            Logger.error('Recorder: Cannot start recording while playing.')
            return
        recorder.record = not recorder.record
    elif key == 288:
        if recorder.record:
            Logger.error('Recorder: Cannot start playing while recording.')
            return
        recorder.play = not recorder.play
    elif key == 287:
        if recorder.play:
            recorder.unbind(play=replay)
        else:
            recorder.bind(play=replay)
            recorder.play = True