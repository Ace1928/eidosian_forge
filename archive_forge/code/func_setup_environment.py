import snake
import apple
import search
import logging
from typing import List, Optional, Tuple
import pygame as pg
from pygame.math import Vector2
import numpy as np
from random import randint
def setup_environment() -> Tuple[pg.Surface, snake.Snake, apple.Apple, search.SearchAlgorithm, pg.time.Clock]:
    """
    Initializes the game environment, setting up the display, and instantiating game objects.
    Returns the screen, snake, apple, search algorithm instance, and the clock for controlling frame rate.
    """
    pg.init()
    screen: pg.Surface = pg.display.set_mode(screen_size)
    snake_entity: snake.Snake = snake_instance
    apple_entity: apple.Apple = apple_instance
    search_entity: search.SearchAlgorithm = search_algorithm
    search_entity.get_path()
    clock: pg.time.Clock = game_clock
    logging.info('Game environment setup complete')
    return (screen, snake_entity, apple_entity, search_entity, clock)