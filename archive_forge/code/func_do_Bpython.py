import pdb
import bpython
def do_Bpython(self, arg):
    locals_ = self.curframe.f_globals.copy()
    locals_.update(self.curframe.f_locals)
    bpython.embed(locals_, ['-i'])