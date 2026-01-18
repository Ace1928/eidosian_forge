from copy import deepcopy
def get_indiv(line):
    """Extract the details of the individual information on the line."""

    def int_no_zero(val):
        """Return integer of val, or None if is zero."""
        v = int(val)
        if v == 0:
            return None
        return v
    indiv_name, marker_line = line.split(',')
    markers = marker_line.replace('\t', ' ').split(' ')
    markers = [marker for marker in markers if marker != '']
    if len(markers[0]) in [2, 4]:
        marker_len = 2
    else:
        marker_len = 3
    try:
        allele_list = [(int_no_zero(marker[0:marker_len]), int_no_zero(marker[marker_len:])) for marker in markers]
    except ValueError:
        allele_list = [(int_no_zero(marker[0:marker_len]),) for marker in markers]
    return (indiv_name, allele_list, marker_len)