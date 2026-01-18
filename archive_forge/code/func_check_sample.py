import unittest
from numpy import int8, int16, uint8, uint16, float32, array, alltrue
import pygame
import pygame.sndarray
def check_sample(size, channels, test_data):
    try:
        pygame.mixer.init(22050, size, channels, allowedchanges=0)
    except pygame.error:
        return
    try:
        __, sz, __ = pygame.mixer.get_init()
        if sz == size:
            zeroed = null_byte * (abs(size) // 8 * len(test_data) * channels)
            snd = pygame.mixer.Sound(buffer=zeroed)
            samples = pygame.sndarray.samples(snd)
            self._assert_compatible(samples, size)
            samples[...] = test_data
            arr = pygame.sndarray.array(snd)
            self.assertTrue(alltrue(samples == arr), 'size: %i\n%s\n%s' % (size, arr, test_data))
    finally:
        pygame.mixer.quit()