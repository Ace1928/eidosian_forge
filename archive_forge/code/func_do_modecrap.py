from . import screen
from . import FSM
import string
def do_modecrap(self, fsm):
    """Handler for \x1b[?<number>h and \x1b[?<number>l. If anyone
        wanted to actually use these, they'd need to add more states to the
        FSM rather than just improve or override this method. """
    screen = fsm.memory[0]
    fsm.memory = [screen]