import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def get_augment_values(self, filename: str) -> Iterator[AugmentValues]:
    yield AugmentValues.default()
    rng = random.Random(self.seed + hash(filename))
    for _ in range(int(self.augment_data_factor - 1)):
        randomized_pools = [list(pool) for pool in self._mixup_pools]
        for pool in randomized_pools:
            rng.shuffle(pool)
        instrument_bin_remap = {}
        for i, pool in enumerate(randomized_pools):
            for j, instrument in enumerate(pool):
                instrument_bin_remap[instrument] = randomized_pools[i - 1][j]
        yield AugmentValues(instrument_bin_remap=instrument_bin_remap, velocity_mod_factor=1.0 + rng.choice(self.velocity_mod_pct), transpose_semitones=rng.choice(self.transpose_semitones), time_stretch_factor=1.0 + rng.choice(self.time_stretch_pct))