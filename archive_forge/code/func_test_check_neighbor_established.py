import time
from os_ken.tests.integrated.common import docker_base as ctn_base
from . import base
def test_check_neighbor_established(self):
    neighbor_state = ctn_base.BGP_FSM_IDLE
    for _ in range(0, self.checktime):
        neighbor_state = self.q1.get_neighbor_state(self.r1)
        if neighbor_state == ctn_base.BGP_FSM_ESTABLISHED:
            break
        time.sleep(1)
    self.assertEqual(neighbor_state, ctn_base.BGP_FSM_ESTABLISHED)