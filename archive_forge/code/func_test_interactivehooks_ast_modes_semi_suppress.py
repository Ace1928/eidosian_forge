import sys
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import CapturedIO
def test_interactivehooks_ast_modes_semi_suppress():
    """
    Test that ast nodes can be triggered with different modes and suppressed
    by semicolon
    """
    saved_mode = ip.ast_node_interactivity
    ip.ast_node_interactivity = 'last_expr_or_assign'
    try:
        with AssertNotPrints('2'):
            ip.run_cell('x = 1+1;', store_history=True)
        with AssertNotPrints('7'):
            ip.run_cell('y = 1+6; # comment with a semicolon', store_history=True)
        with AssertNotPrints('9'):
            ip.run_cell('z = 1+8;\n#commented_out_function()', store_history=True)
    finally:
        ip.ast_node_interactivity = saved_mode