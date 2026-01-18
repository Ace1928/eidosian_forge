import os
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.shape import ShapeRect
from kivy.input.motionevent import MotionEvent
def create_joystick(self, index):
    Logger.info('Android: create joystick <%d>' % index)
    js = pygame.joystick.Joystick(index)
    js.init()
    if js.get_numbuttons() == 0:
        Logger.info('Android: discard joystick <%d> cause no button' % index)
        return
    self.joysticks.append(js)