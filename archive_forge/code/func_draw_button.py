import pygame
from Constants import *
from GA import *
import sys
def draw_button(self):
    action = False
    pos = pygame.mouse.get_pos()
    button_rect = pygame.Rect(self.x, self.y, BTN_WIDTH, BTN_HEIGHT)
    if button_rect.collidepoint(pos):
        if pygame.mouse.get_pressed()[0] == 1:
            self.clicked = True
            pygame.draw.rect(self.game.display, BTN_CLICKED, button_rect)
        elif pygame.mouse.get_pressed()[0] == 0 and self.clicked == True:
            self.clicked = False
            action = True
        else:
            pygame.draw.rect(self.game.display, BTN_HOVER, button_rect)
    else:
        pygame.draw.rect(self.game.display, BTN_COLOR, button_rect)
    text_img = self.font.render(self.text, True, WHITE)
    text_len = text_img.get_width()
    self.game.display.blit(text_img, (self.x + int(BTN_WIDTH / 2) - int(text_len / 2), self.y + 25))
    return action