from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.clock import Clock
import timeit
def stress_insert(self, *largs):
    self.test_done = False
    text_input = self.text_input
    text_input.select_all()
    text_input.copy(text_input.selection_text)
    text_input.cursor = text_input.get_cursor_from_index(text_input.selection_to)
    len_text = len(text_input._lines)
    self.tot_time = 0
    ev = None

    def pste(*l):
        if len(text_input._lines) >= len_text * 9:
            ev.cancel()
            print('Done!')
            m_len = len(text_input._lines)
            print('pasted', len_text, 'lines', round((m_len - len_text) / len_text), 'times')
            import resource
            print('mem usage after test')
            print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'MB')
            print('total lines in text input:', m_len)
            print('--------------------------------------')
            print('total time elapsed:', self.tot_time)
            print('--------------------------------------')
            self.test_done = True
            return
        self.tot_time += l[0]
        text_input.paste()
        ev()
    ev = Clock.create_trigger(pste)
    ev()