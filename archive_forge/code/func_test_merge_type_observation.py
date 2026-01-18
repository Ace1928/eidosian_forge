from minerl.herobraine.hero.handlers.agent.observations.compass import CompassObservation
from minerl.herobraine.hero.handlers.agent.observations.inventory import FlatInventoryObservation
from minerl.herobraine.hero.handlers.agent.observations.equipped_item import _TypeObservation
from minerl.herobraine.hero.handlers.agent.action import ItemListAction
def test_merge_type_observation():
    type_obs_a = _TypeObservation('test', ['none', 'A', 'B', 'C', 'D', 'other'], _default='none', _other='other')
    type_obs_b = _TypeObservation('test', ['none', 'E', 'F', 'other'], _default='none', _other='other')
    type_obs_result = _TypeObservation('test', ['none', 'A', 'B', 'C', 'D', 'E', 'F', 'other'], _default='none', _other='other')
    assert type_obs_a | type_obs_b == type_obs_result