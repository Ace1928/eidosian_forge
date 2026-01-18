from nltk.translate.ribes_score import corpus_ribes, word_rank_alignment
def test_no_zero_div():
    hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
    ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
    ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']
    hyp2 = ['he', 'read', 'the']
    ref2a = ['he', 'was', 'interested', 'in', 'world', 'history', 'because', 'he']
    list_of_refs = [[ref1a, ref1b, ref1c], [ref2a]]
    hypotheses = [hyp1, hyp2]
    score = corpus_ribes(list_of_refs, hypotheses)
    assert round(score, 4) == 0.1688