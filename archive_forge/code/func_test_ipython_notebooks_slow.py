import os
from pathlib import Path
import pytest
@pytest.mark.skip(reason='these notebooks need special software or hardware')
@pytest.mark.parametrize('nb_file', ('examples/00_intro_to_thinc.ipynb', 'examples/02_transformers_tagger_bert.ipynb', 'examples/03_pos_tagger_basic_cnn.ipynb', 'examples/03_textcat_basic_neural_bow.ipynb', 'examples/04_configure_gpu_memory.ipynb', 'examples/04_parallel_training_ray.ipynb', 'examples/05_visualizing_models.ipynb', 'examples/06_predicting_like_terms.ipynb'))
def test_ipython_notebooks_slow(test_files: None):
    ...