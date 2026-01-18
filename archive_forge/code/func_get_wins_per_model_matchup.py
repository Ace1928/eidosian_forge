import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.stats import binom_test
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai import __file__ as parlai_filepath
from parlai.core.params import ParlaiParser
import json
from IPython.core.display import HTML
def get_wins_per_model_matchup(self) -> pd.DataFrame:
    """
        Return the wins for each model by matchup.
        """
    df_filtered = self.filter_by_dialogue_length(True)
    self.matchup_total_df = df_filtered.groupby(['eval_choice_0', 'eval_choice_1'])['run_id'].count().to_frame('matchup_total')
    self.win_total_df = df_filtered.groupby(['eval_choice_0', 'eval_choice_1', 'winner', 'loser'])['loser'].count().to_frame('win_total').reset_index().set_index(['eval_choice_0', 'eval_choice_1'])
    return self.win_total_df