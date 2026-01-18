import pygame
from Constants import *
from GA import *
import sys
def display_menu(self):
    self.run_display = True
    while self.run_display:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game.running, self.game.playing = (False, False)
                self.game.curr_menu.run_display = False
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.game.BACK = True
                if self.no_population.active:
                    if event.key == pygame.K_BACKSPACE:
                        self.no_population.input = self.no_population.input[:-1]
                    elif event.unicode.isdigit() and len(self.no_population.input) < 3:
                        self.no_population.input += event.unicode
                if self.no_generation.active:
                    if event.key == pygame.K_BACKSPACE:
                        self.no_generation.input = self.no_generation.input[:-1]
                    elif event.unicode.isdigit() and len(self.no_generation.input) < 3:
                        self.no_generation.input += event.unicode
                if self.no_hidden_nodes.active:
                    if event.key == pygame.K_BACKSPACE:
                        self.no_hidden_nodes.input = self.no_hidden_nodes.input[:-1]
                    elif event.unicode.isdigit() and len(self.no_hidden_nodes.input) < 2:
                        self.no_hidden_nodes.input += event.unicode
                if self.mutation_rate.active:
                    if event.key == pygame.K_BACKSPACE:
                        self.mutation_rate.input = self.mutation_rate.input[:-1]
                    elif event.unicode.isdigit() and len(self.mutation_rate.input) < 3:
                        self.mutation_rate.input += event.unicode
        self.check_input()
        self.game.display.fill(WINDOW_COLOR)
        self.game.draw_text('GA Options', self.title_size, self.game.SIZE / 2, self.game.SIZE / 2 - 4 * (CELL_SIZE + NO_OF_CELLS), color=TITLE_COLOR)
        self.game.draw_text('Settings to train model:', 25, self.game.SIZE / 2, self.game.SIZE / 2 - 2 * (CELL_SIZE + NO_OF_CELLS), color=MENU_COLOR)
        self.game.draw_text('No. of populations     : ', 20, self.game.SIZE / 2 - 2 * CELL_SIZE, self.game.SIZE / 2 - 50, color=BANNER_COLOR)
        self.game.draw_text('No. of generations     : ', 20, self.game.SIZE / 2 - 2 * CELL_SIZE, self.game.SIZE / 2, color=BANNER_COLOR)
        self.game.draw_text('No. of hidden nodes   : ', 20, self.game.SIZE / 2 - 2 * CELL_SIZE, self.game.SIZE / 2 + 50, color=BANNER_COLOR)
        self.game.draw_text('Mutation rate %:          : ', 20, self.game.SIZE / 2 - 2 * CELL_SIZE, self.game.SIZE / 2 + 100, color=BANNER_COLOR)
        self.no_population.draw_input()
        self.no_generation.draw_input()
        self.no_hidden_nodes.draw_input()
        self.mutation_rate.draw_input()
        if self.load_model.draw_button():
            self.load_GA()
        if self.train_model.draw_button():
            self.train_GA()
        self.game.draw_text('Q to return to main menu', 20, self.game.SIZE / 2, self.game.SIZE / 2 + 6 * (NO_OF_CELLS + CELL_SIZE), color=WHITE)
        self.blit_menu()
    self.reset()