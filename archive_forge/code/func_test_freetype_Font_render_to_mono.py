import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_render_to_mono(self):
    font = self._TEST_FONTS['sans']
    text = ' .'
    rect = font.get_rect(text, size=24)
    size = rect.size
    fg = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
    bg = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
    surrogate = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
    surfaces = [pygame.Surface(size, 0, 8), pygame.Surface(size, 0, 16), pygame.Surface(size, pygame.SRCALPHA, 16), pygame.Surface(size, 0, 24), pygame.Surface(size, 0, 32), pygame.Surface(size, pygame.SRCALPHA, 32)]
    fg_colors = [surfaces[0].get_palette_at(2), surfaces[1].unmap_rgb(surfaces[1].map_rgb((128, 64, 200))), surfaces[2].unmap_rgb(surfaces[2].map_rgb((99, 0, 100, 64))), (128, 97, 213), (128, 97, 213), (128, 97, 213, 60)]
    fg_colors = [pygame.Color(*c) for c in fg_colors]
    self.assertEqual(len(surfaces), len(fg_colors))
    bg_colors = [surfaces[0].get_palette_at(4), surfaces[1].unmap_rgb(surfaces[1].map_rgb((220, 20, 99))), surfaces[2].unmap_rgb(surfaces[2].map_rgb((55, 200, 0, 86))), (255, 120, 13), (255, 120, 13), (255, 120, 13, 180)]
    bg_colors = [pygame.Color(*c) for c in bg_colors]
    self.assertEqual(len(surfaces), len(bg_colors))
    save_antialiased = font.antialiased
    font.antialiased = False
    try:
        fill_color = pygame.Color('black')
        for i, surf in enumerate(surfaces):
            surf.fill(fill_color)
            fg_color = fg_colors[i]
            fg.set_at((0, 0), fg_color)
            surf.blit(fg, (0, 0))
            r_fg_color = surf.get_at((0, 0))
            surf.set_at((0, 0), fill_color)
            rrect = font.render_to(surf, (0, 0), text, fg_color, size=24)
            bottomleft = (0, rrect.height - 1)
            self.assertEqual(surf.get_at(bottomleft), fill_color, 'Position: {}. Depth: {}. fg_color: {}.'.format(bottomleft, surf.get_bitsize(), fg_color))
            bottomright = (rrect.width - 1, rrect.height - 1)
            self.assertEqual(surf.get_at(bottomright), r_fg_color, 'Position: {}. Depth: {}. fg_color: {}.'.format(bottomright, surf.get_bitsize(), fg_color))
        for i, surf in enumerate(surfaces):
            surf.fill(fill_color)
            fg_color = fg_colors[i]
            bg_color = bg_colors[i]
            bg.set_at((0, 0), bg_color)
            fg.set_at((0, 0), fg_color)
            if surf.get_bitsize() == 24:
                surrogate.set_at((0, 0), fill_color)
                surrogate.blit(bg, (0, 0))
                r_bg_color = surrogate.get_at((0, 0))
                surrogate.blit(fg, (0, 0))
                r_fg_color = surrogate.get_at((0, 0))
            else:
                surf.blit(bg, (0, 0))
                r_bg_color = surf.get_at((0, 0))
                surf.blit(fg, (0, 0))
                r_fg_color = surf.get_at((0, 0))
                surf.set_at((0, 0), fill_color)
            rrect = font.render_to(surf, (0, 0), text, fg_color, bg_color, size=24)
            bottomleft = (0, rrect.height - 1)
            self.assertEqual(surf.get_at(bottomleft), r_bg_color)
            bottomright = (rrect.width - 1, rrect.height - 1)
            self.assertEqual(surf.get_at(bottomright), r_fg_color)
    finally:
        font.antialiased = save_antialiased