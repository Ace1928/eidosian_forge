import unittest
import torch
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.extras import BestOfNSampler
def test_different_sample_sizes_and_n_candidates_values(self):
    """
        Tests different sample sizes and n_candidates values
        """
    generation_config = GenerationConfig(min_length=-1, top_k=0.0, top_p=1.0, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
    output_length_sampler = LengthSampler(6, 10)
    for sample_value, n_candidates_values, expected in [(4, 2, 2), (10, 3, 3), (6, 4, 4)]:
        best_of_n = BestOfNSampler(self.model, self.tokenizer, queries_to_scores, length_sampler=output_length_sampler, generation_config=generation_config, sample_size=sample_value, n_candidates=n_candidates_values)
        queries = ['hello world', 'troll the world']
        tokenized_queries = [self.tokenizer.encode(query) for query in queries]
        results = best_of_n.generate(tokenized_queries)
        for result in results:
            assert len(result) == expected