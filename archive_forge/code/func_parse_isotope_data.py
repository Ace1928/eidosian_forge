from urllib import request
def parse_isotope_data(raw_data):
    indexes = [idx for idx, line in enumerate(raw_data) if '_____' in line]
    isotopes = {}
    for idx1, idx2 in zip(indexes, indexes[1:]):
        atomic_number = int(raw_data[idx1 + 1].split()[0])
        isotopes[atomic_number] = dct = {}
        for isotope_idx in range(idx1 + 1, idx2):
            mass_number = int(raw_data[isotope_idx][8:12])
            mass = float(raw_data[isotope_idx][13:31].split('(')[0])
            try:
                composition = float(raw_data[isotope_idx][32:46].split('(')[0])
            except ValueError:
                composition = 0.0
            dct[mass_number] = {'mass': mass, 'composition': composition}
    return isotopes