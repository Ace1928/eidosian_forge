from .gui import *
from . import smooth
from .colors import Palette
from .manager import LinkManager
def build_save_image_menu(self, menubar, parent_menu):
    menu = self.save_image_menu = Tk_.Menu(menubar, tearoff=0)
    save = self.save_image
    for item_name, save_function in [('PostScript (color)', lambda: save('eps', 'color')), ('PostScript (grays)', lambda: save('eps', 'gray')), ('SVG', lambda: save('svg', 'color')), ('TikZ', lambda: save('tikz', 'color')), ('PDF', lambda: save('pdf', 'color'))]:
        menu.add_command(label=item_name, command=save_function)
    self.disable_fancy_save_images()
    self.enable_fancy_save_images()
    parent_menu.add_cascade(label='Save Image...', menu=menu)