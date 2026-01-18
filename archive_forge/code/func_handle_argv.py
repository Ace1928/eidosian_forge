import os
import sys
def handle_argv(self, argv, i, setup):
    assert argv[i] == self.arg_v_rep
    del argv[i]
    setup[self.arg_name] = True