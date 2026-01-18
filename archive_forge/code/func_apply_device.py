import sys
from os import environ
from kivy.config import Config
from kivy.logger import Logger
def apply_device(device, scale, orientation):
    name, width, height, dpi, density = devices[device]
    if orientation == 'portrait':
        width, height = (height, width)
    Logger.info('Screen: Apply screen settings for {0}'.format(name))
    Logger.info('Screen: size={0}x{1} dpi={2} density={3} orientation={4}'.format(width, height, dpi, density, orientation))
    try:
        scale = float(scale)
    except:
        scale = 1
    environ['KIVY_METRICS_DENSITY'] = str(density * scale)
    environ['KIVY_DPI'] = str(dpi * scale)
    Config.set('graphics', 'width', str(int(width * scale)))
    Config.set('graphics', 'height', str(int(height * scale - 25 * density)))
    Config.set('graphics', 'fullscreen', '0')
    Config.set('graphics', 'show_mousecursor', '1')