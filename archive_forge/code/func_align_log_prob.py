import math
def align_log_prob(i, j, source_sents, target_sents, alignment, params):
    """Returns the log probability of the two sentences C{source_sents[i]}, C{target_sents[j]}
    being aligned with a specific C{alignment}.

    @param i: The offset of the source sentence.
    @param j: The offset of the target sentence.
    @param source_sents: The list of source sentence lengths.
    @param target_sents: The list of target sentence lengths.
    @param alignment: The alignment type, a tuple of two integers.
    @param params: The sentence alignment parameters.

    @returns: The log probability of a specific alignment between the two sentences, given the parameters.
    """
    l_s = sum((source_sents[i - offset - 1] for offset in range(alignment[0])))
    l_t = sum((target_sents[j - offset - 1] for offset in range(alignment[1])))
    try:
        m = (l_s + l_t / params.AVERAGE_CHARACTERS) / 2
        delta = (l_s * params.AVERAGE_CHARACTERS - l_t) / math.sqrt(m * params.VARIANCE_CHARACTERS)
    except ZeroDivisionError:
        return float('-inf')
    return -(LOG2 + norm_logsf(abs(delta)) + math.log(params.PRIORS[alignment]))