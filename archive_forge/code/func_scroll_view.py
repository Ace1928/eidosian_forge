import sys
import os
import pygame as pg
from pygame.transform import scale
def scroll_view(screen, image: pg.Surface, direction: int, view_rect):
    src_rect = None
    dst_rect = None
    zoom_view_rect = screen.get_clip()
    image_w, image_h = image.get_size()
    if direction == DIR_UP:
        if view_rect.top > 0:
            screen.scroll(dy=zoom_factor)
            view_rect.move_ip(0, -1)
            src_rect = view_rect.copy()
            src_rect.h = 1
            dst_rect = zoom_view_rect.copy()
            dst_rect.h = zoom_factor
    elif direction == DIR_DOWN:
        if view_rect.bottom < image_h:
            screen.scroll(dy=-zoom_factor)
            view_rect.move_ip(0, 1)
            src_rect = view_rect.copy()
            src_rect.h = 1
            src_rect.bottom = view_rect.bottom
            dst_rect = zoom_view_rect.copy()
            dst_rect.h = zoom_factor
            dst_rect.bottom = zoom_view_rect.bottom
    elif direction == DIR_LEFT:
        if view_rect.left > 0:
            screen.scroll(dx=zoom_factor)
            view_rect.move_ip(-1, 0)
            src_rect = view_rect.copy()
            src_rect.w = 1
            dst_rect = zoom_view_rect.copy()
            dst_rect.w = zoom_factor
    elif direction == DIR_RIGHT:
        if view_rect.right < image_w:
            screen.scroll(dx=-zoom_factor)
            view_rect.move_ip(1, 0)
            src_rect = view_rect.copy()
            src_rect.w = 1
            src_rect.right = view_rect.right
            dst_rect = zoom_view_rect.copy()
            dst_rect.w = zoom_factor
            dst_rect.right = zoom_view_rect.right
    if src_rect is not None and dst_rect is not None:
        scale(image.subsurface(src_rect), dst_rect.size, screen.subsurface(dst_rect))
        pg.display.update(zoom_view_rect)