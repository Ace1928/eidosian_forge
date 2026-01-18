from .adaptation import OpenALDriver
import pyglet
def create_audio_driver(device_name=None):
    _driver = OpenALDriver(device_name)
    if _debug:
        print('OpenAL', _driver.get_version())
    return _driver