from minerl.herobraine.hero.handlers.agent.action import Action
import jinja2
import minerl.herobraine.hero.spaces as spaces
import numpy as np
def from_universal(self, x):
    if 'custom_action' in x and 'cameraYaw' in x['custom_action'] and ('cameraPitch' in x['custom_action']):
        delta_pitch = x['custom_action']['cameraPitch']
        delta_yaw = x['custom_action']['cameraYaw']
        assert not np.isnan(np.sum(x['custom_action']['cameraYaw'])), 'NAN in action!'
        assert not np.isnan(np.sum(x['custom_action']['cameraPitch'])), 'NAN in action!'
        return np.array([-delta_pitch, -delta_yaw], dtype=np.float32)
    else:
        return np.array([0.0, 0.0], dtype=np.float32)