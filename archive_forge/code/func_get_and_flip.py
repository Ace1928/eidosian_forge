import pygame as pg
import pygame.camera
def get_and_flip(self):
    self.snapshot = self.camera.get_image(self.display)
    pg.display.flip()