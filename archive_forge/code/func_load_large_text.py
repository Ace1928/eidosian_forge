from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.clock import Clock
import timeit
def load_large_text(self, *largs):
    print('loading uix/textinput.py....')
    self.test_done = False
    fd = open(resource_find('uix/textinput.py'), 'r')
    print('putting text in textinput')

    def load_text(*l):
        self.text_input.text = fd.read()
    t = timeit.Timer(load_text)
    ttk = t.timeit(1)
    fd.close()
    import resource
    print('mem usage after test')
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'MB')
    print('------------------------------------------')
    print('Loaded', len(self.text_input._lines), 'lines', ttk, 'secs')
    print('------------------------------------------')
    self.test_done = True