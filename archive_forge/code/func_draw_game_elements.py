import snake
import apple
import search
import logging
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
def draw_game_elements(screen: pg.Surface, snake_entity: snake.Snake, apple_entity: apple.Apple, clock: pg.time.Clock) -> None:
    """
    Manages the drawing of all game objects on the screen.
    This function separates game logic updates from rendering for modularity and flexibility.
    """
    screen.fill((51, 51, 51))
    pg.draw.rect(screen, border_color, pg.Rect(0, 0, screen_size[0] + 2 * border_width, screen_size[1] + 2 * border_width), border_width)
    snake_entity.show(screen)
    apple_entity.show(screen)
    snake_entity.update(apple_entity)
    pg.display.flip()
    clock.tick(frames_per_second)
    logging.debug('Game elements drawn on the screen')