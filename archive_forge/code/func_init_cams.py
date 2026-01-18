import pygame as pg
import pygame.camera
def init_cams(self, which_cam_idx):
    self.clist = pygame.camera.list_cameras()
    if not self.clist:
        raise ValueError('Sorry, no cameras detected.')
    try:
        cam_id = self.clist[which_cam_idx]
    except IndexError:
        cam_id = self.clist[0]
    self.camera = pygame.camera.Camera(cam_id, self.size, 'RGB')
    self.camera.start()
    self.clock = pg.time.Clock()
    self.snapshot = pg.surface.Surface(self.size, 0, self.display)
    return cam_id