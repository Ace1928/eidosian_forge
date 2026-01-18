import sys
import os
import pygame as pg
def render_fonts(self, text='A display of font &N'):
    """
        Build a list that includes a surface and the running total of their
        height for each font in the font list. Store the largest width and
        other variables for later use.
        """
    font_size = self.font_size
    color = (255, 255, 255)
    instruction_color = (255, 255, 0)
    self.back_color = (0, 0, 0)
    fonts, path = self.get_font_list()
    font_surfaces = []
    total_height = 0
    max_width = 0
    load_font = pg.font.Font if path else pg.font.SysFont
    font = pg.font.SysFont(pg.font.get_default_font(), font_size)
    lines = ('Use the scroll wheel or click and drag', 'to scroll up and down.', "Fonts that don't use the Latin Alphabet", 'might render incorrectly.', f'Here are your {len(fonts)} fonts', '')
    for line in lines:
        surf = font.render(line, 1, instruction_color, self.back_color)
        font_surfaces.append((surf, total_height))
        total_height += surf.get_height()
        max_width = max(max_width, surf.get_width())
    for name in sorted(fonts):
        try:
            font = load_font(path + name, font_size)
        except OSError:
            continue
        line = text.replace('&N', name)
        try:
            surf = font.render(line, 1, color, self.back_color)
        except pg.error as e:
            print(e)
            break
        max_width = max(max_width, surf.get_width())
        font_surfaces.append((surf, total_height))
        total_height += surf.get_height()
    self.total_height = total_height
    self.max_width = max_width
    self.font_surfaces = font_surfaces
    self.max_y = total_height - pg.display.get_surface().get_height()