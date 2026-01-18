import statemachine
import trafficlightstate
def change(self):
    self._state = self._state.next_state()