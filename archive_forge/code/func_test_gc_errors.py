from pyomo.common.gc_manager import PauseGC
import gc
import pyomo.common.unittest as unittest
def test_gc_errors(self):
    pgc = PauseGC()
    self.assertTrue(gc.isenabled())
    with pgc:
        with self.assertRaisesRegex(RuntimeError, 'Entering PauseGC context manager that was already entered'):
            with pgc:
                pass
        self.assertFalse(gc.isenabled())
    self.assertTrue(gc.isenabled())
    with pgc:
        self.assertFalse(gc.isenabled())
        with PauseGC():
            self.assertFalse(gc.isenabled())
            with self.assertRaisesRegex(RuntimeError, 'Exiting PauseGC context manager out of order: there are other active PauseGC context managers that were entered after this context manager and have not yet been exited.'):
                pgc.close()
        self.assertFalse(gc.isenabled())
    self.assertTrue(gc.isenabled())