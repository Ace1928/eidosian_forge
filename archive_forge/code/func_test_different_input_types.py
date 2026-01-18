import unittest
import torch
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.extras import BestOfNSampler
def test_different_input_types(self):
    """
        Tests if the different input types normalizer works
        """
    generation_config = GenerationConfig(min_length=-1, top_k=0.0, top_p=1.0, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
    output_length_sampler = LengthSampler(2, 6)
    best_of_n = BestOfNSampler(self.model, self.tokenizer, queries_to_scores, length_sampler=output_length_sampler, generation_config=generation_config)
    queries = ['hello world', 'goodbye world']
    tokenized_queries = [self.tokenizer.encode(query) for query in queries]
    various_queries_formats = [(tokenized_queries[0], 1), (tokenized_queries, 2), (torch.tensor(tokenized_queries[1]), 1), ([torch.tensor(query) for query in tokenized_queries], 2)]
    for q, expected_length in various_queries_formats:
        results = best_of_n.generate(q)
        assert isinstance(results, list)
        assert len(results) == expected_length