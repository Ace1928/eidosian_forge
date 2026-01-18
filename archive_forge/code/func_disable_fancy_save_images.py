from .gui import *
from . import smooth
from .colors import Palette
from .manager import LinkManager
def disable_fancy_save_images(self):
    for i in [3, 4]:
        self.save_image_menu.entryconfig(i, state='disabled')