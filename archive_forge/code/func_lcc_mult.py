import math
def lcc_mult(seq, wsize):
    """Calculate Local Composition Complexity (LCC) values over sliding window.

    Returns a list of floats, the LCC values for a sliding window over
    the sequence.

    seq - an unambiguous DNA sequence (a string or Seq object)
    wsize - window size, integer

    The result is the same as applying lcc_simp multiple times, but this
    version is optimized for speed. The optimization works by using the
    value of previous window as a base to compute the next one.
    """
    l4 = math.log(4)
    seq = seq.upper()
    tamseq = len(seq)
    compone = [0]
    lccsal = []
    for i in range(wsize):
        compone.append((i + 1) / wsize * math.log((i + 1) / wsize) / l4)
    window = seq[0:wsize]
    cant_a = window.count('A')
    cant_c = window.count('C')
    cant_t = window.count('T')
    cant_g = window.count('G')
    term_a = compone[cant_a]
    term_c = compone[cant_c]
    term_t = compone[cant_t]
    term_g = compone[cant_g]
    lccsal.append(-(term_a + term_c + term_t + term_g))
    tail = seq[0]
    for x in range(tamseq - wsize):
        window = seq[x + 1:wsize + x + 1]
        if tail == window[-1]:
            lccsal.append(lccsal[-1])
        elif tail == 'A':
            cant_a -= 1
            if window.endswith('C'):
                cant_c += 1
                term_a = compone[cant_a]
                term_c = compone[cant_c]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('T'):
                cant_t += 1
                term_a = compone[cant_a]
                term_t = compone[cant_t]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('G'):
                cant_g += 1
                term_a = compone[cant_a]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
        elif tail == 'C':
            cant_c -= 1
            if window.endswith('A'):
                cant_a += 1
                term_a = compone[cant_a]
                term_c = compone[cant_c]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('T'):
                cant_t += 1
                term_c = compone[cant_c]
                term_t = compone[cant_t]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('G'):
                cant_g += 1
                term_c = compone[cant_c]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
        elif tail == 'T':
            cant_t -= 1
            if window.endswith('A'):
                cant_a += 1
                term_a = compone[cant_a]
                term_t = compone[cant_t]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('C'):
                cant_c += 1
                term_c = compone[cant_c]
                term_t = compone[cant_t]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('G'):
                cant_g += 1
                term_t = compone[cant_t]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
        elif tail == 'G':
            cant_g -= 1
            if window.endswith('A'):
                cant_a += 1
                term_a = compone[cant_a]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('C'):
                cant_c += 1
                term_c = compone[cant_c]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
            elif window.endswith('T'):
                cant_t += 1
                term_t = compone[cant_t]
                term_g = compone[cant_g]
                lccsal.append(-(term_a + term_c + term_t + term_g))
        tail = window[0]
    return lccsal