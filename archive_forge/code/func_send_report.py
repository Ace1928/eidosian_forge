import os
import sys
import platform as plf
from time import ctime
from configparser import ConfigParser
from io import StringIO
import kivy
from kivy.core import gl
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from kivy.core.camera import Camera
from kivy.core.image import ImageLoader
from kivy.core.text import Label
from kivy.core.video import Video
from kivy.config import Config
from kivy.input.factory import MotionEventFactory
def send_report(dict_report):
    import requests
    import json
    gist_report = {'description': 'Report', 'public': 'true', 'files': {'Global.txt': {'content': '\n'.join(dict_report['Global']), 'type': 'text'}, 'OpenGL.txt': {'content': '\n'.join(dict_report['OpenGL']), 'type': 'text'}, 'Core selection.txt': {'content': '\n'.join(dict_report['Core']), 'type': 'text'}, 'Libraries.txt': {'content': '\n'.join(dict_report['Libraries']), 'type': 'text'}, 'Configuration.txt': {'content': '\n'.join(dict_report['Configuration']), 'type': 'text'}, 'Input Availability.txt': {'content': '\n'.join(dict_report['InputAvailability']), 'type': 'text'}, 'Environ.txt': {'content': '\n'.join(dict_report['Environ']), 'type': 'text'}, 'Options.txt': {'content': '\n'.join(dict_report['Options']), 'type': 'text'}}}
    report_json = json.dumps(gist_report)
    response = requests.post('https://api.github.com/gists', report_json)
    return json.loads(response.text)['html_url']