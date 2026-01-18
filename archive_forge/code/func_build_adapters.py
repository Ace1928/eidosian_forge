import unittest
from zope.interface.tests import OptimizationTestMixin
def build_adapters(L, MT):
    return L([MT({IR0: MT({'': 'A1', 'name1': 'A2'})}), MT({IB0: MT({IR0: MT({'': 'A1', 'name2': 'A2'}), IR1: MT({'': 'A3', 'name3': 'A4'})})}), MT({IB0: MT({IB1: MT({IR0: MT({'': 'A1'})}), IB2: MT({IR0: MT({'name2': 'A2'}), IR1: MT({'name4': 'A4'})}), IB3: MT({IR1: MT({'': 'A3'})})})})])