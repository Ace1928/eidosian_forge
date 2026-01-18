import sys
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import CapturedIO
def test_interactivehooks_ast_modes():
    """
    Test that ast nodes can be triggered with different modes
    """
    saved_mode = ip.ast_node_interactivity
    ip.ast_node_interactivity = 'last_expr_or_assign'
    try:
        with AssertPrints('2'):
            ip.run_cell('a = 1+1', store_history=True)
        with AssertPrints('9'):
            ip.run_cell('b = 1+8 # comment with a semicolon;', store_history=False)
        with AssertPrints('7'):
            ip.run_cell('c = 1+6\n#commented_out_function();', store_history=True)
        ip.run_cell('d = 11', store_history=True)
        with AssertPrints('12'):
            ip.run_cell('d += 1', store_history=True)
        with AssertNotPrints('42'):
            ip.run_cell('(u,v) = (41+1, 43-1)')
    finally:
        ip.ast_node_interactivity = saved_mode