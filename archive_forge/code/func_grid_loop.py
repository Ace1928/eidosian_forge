import pygame as pg
def grid_loop(self):
    self.surface.fill((0, 0, 0))
    for row in range(TILES_HORIZONTAL):
        for col in range(row % 2, TILES_HORIZONTAL, 2):
            pg.draw.rect(self.surface, (40, 40, 40), (row * TILE_SIZE, col * TILE_SIZE, TILE_SIZE, TILE_SIZE))
    self.player.draw()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            self.loop = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self.loop = False
        elif event.type == pg.MOUSEBUTTONUP:
            pos = pg.mouse.get_pos()
            self.player.move(pos)
    pg.display.update()