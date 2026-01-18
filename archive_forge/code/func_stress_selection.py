from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.clock import Clock
import timeit
def stress_selection(self, *largs):
    self.test_done = False
    text_input = self.text_input
    self.tot_time = 0
    old_selection_from = text_input.selection_from - 210
    ev = None

    def pste(*l):
        if text_input.selection_from >= old_selection_from:
            ev.cancel()
            print('Done!')
            import resource
            print('mem usage after test')
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'MB')
            print('--------------------------------------')
            print('total time elapsed:', self.tot_time)
            print('--------------------------------------')
            self.test_done = True
            return
        text_input.select_text(text_input.selection_from - 1, text_input.selection_to)
        ev()
    ev = Clock.create_trigger(pste)
    ev()