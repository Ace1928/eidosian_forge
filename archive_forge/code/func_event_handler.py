import pygame
from Constants import *
from Menu import *
from GameController import GameController
from GA import *
import sys
def event_handler(self):
    for event in pygame.event.get():
        if self.is_quit(event):
            print('Bye :)')
            pygame.quit()
            sys.exit()
        elif self.playing and event.type == pygame.USEREVENT:
            if self.load_model:
                self.controller.load_model()
                self.load_model = False
            self.controller.ai_play(self.curr_menu.state)
            if self.controller.end == True:
                self.playing = False
                self.game_over()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.START = True
                self.view_path = False
            elif event.key == pygame.K_q:
                self.BACK = True
                self.controller.reset()
            elif event.key == pygame.K_SPACE:
                self.view_path = not self.view_path
            elif event.key == pygame.K_DOWN:
                self.DOWNKEY = True
            elif event.key == pygame.K_UP:
                self.UPKEY = True
            elif event.key == pygame.K_w:
                self.speed_up = -1 * self.speed_up
                self.speed = self.speed + self.speed_up
                pygame.time.set_timer(self.SCREEN_UPDATE, self.speed)