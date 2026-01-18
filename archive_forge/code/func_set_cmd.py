import io
import sys
import time
import marshal
def set_cmd(self, cmd):
    if self.cur[-1]:
        return
    self.cmd = cmd
    self.simulate_call(cmd)