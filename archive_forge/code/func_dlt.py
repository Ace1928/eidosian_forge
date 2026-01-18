from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.clock import Clock
import timeit
def dlt(*l):
    if len(text_input.text) <= target:
        ev.cancel()
        print('Done!')
        m_len = len(text_input._lines)
        print('deleted 210 characters 9 times')
        import resource
        print('mem usage after test')
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'MB')
        print('total lines in text input:', m_len)
        print('--------------------------------------')
        print('total time elapsed:', self.tot_time)
        print('--------------------------------------')
        self.test_done = True
        return
    text_input.select_text(self.lt - 220, self.lt - 10)
    text_input.delete_selection()
    self.lt -= 210
    text_input.scroll_y -= 100
    self.tot_time += l[0]
    ev()