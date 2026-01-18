import unittest
from zope.interface.tests import OptimizationTestMixin
def build_subscribers(L, F, MT):
    return L([MT({IR0: MT({'': F(['A1', 'A2'])})}), MT({IB0: MT({IR0: MT({'': F(['A1', 'A2'])}), IR1: MT({'': F(['A3', 'A4'])})})}), MT({IB0: MT({IB1: MT({IR0: MT({'': F(['A1'])})}), IB2: MT({IR0: MT({'': F(['A2'])}), IR1: MT({'': F(['A4'])})}), IB3: MT({IR1: MT({'': F(['A3'])})})})})])