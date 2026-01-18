from os.path import exists
from time import time
from kivy.event import EventDispatcher
from kivy.properties import ObjectProperty, BooleanProperty, StringProperty, \
from kivy.input.motionevent import MotionEvent
from kivy.base import EventLoop
from kivy.logger import Logger
from ast import literal_eval
from functools import partial
def on_record(self, instance, value):
    if value:
        self.counter = 0
        self.record_time = time()
        self.record_fd = open(self.filename, 'w')
        self.record_fd.write('#RECORDER1.0\n')
        Logger.info('Recorder: Recording inputs to %r' % self.filename)
    else:
        self.record_fd.close()
        Logger.info('Recorder: Recorded %d events in %r' % (self.counter, self.filename))