from os import remove
from os.path import join
from shutil import copyfile, rmtree
from tempfile import mkdtemp
from threading import Event
from zipfile import ZipFile
from kivy.tests.common import GraphicUnitTest, ensure_web_server
def check_sequence_frames(self, img, frames, slides=5):
    old = None
    while slides:
        self.assertNotEqual(img.anim_index, old)
        old = img.anim_index
        self.advance_frames(frames)
        slides -= 1
    return True