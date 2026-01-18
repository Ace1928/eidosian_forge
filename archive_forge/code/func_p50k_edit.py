from tiktoken.load import data_gym_to_mergeable_bpe_ranks, load_tiktoken_bpe
def p50k_edit():
    mergeable_ranks = load_tiktoken_bpe('https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken', expected_hash='94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069')
    special_tokens = {ENDOFTEXT: 50256, FIM_PREFIX: 50281, FIM_MIDDLE: 50282, FIM_SUFFIX: 50283}
    return {'name': 'p50k_edit', 'pat_str': "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", 'mergeable_ranks': mergeable_ranks, 'special_tokens': special_tokens}