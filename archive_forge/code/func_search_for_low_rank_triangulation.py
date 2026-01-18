import string
from ..sage_helper import _within_sage, sage_method
def search_for_low_rank_triangulation(M, tries=100, target_lower_bound=0):
    rank_lower_bound = max(M.homology().rank(), target_lower_bound)
    rank_upper_bound = M.fundamental_group().num_generators()
    N = M.copy()
    curr_best_tri = N.copy()
    for i in range(tries):
        if rank_upper_bound == rank_lower_bound:
            break
        N.randomize()
        new_rank = N.fundamental_group().num_generators()
        if new_rank < rank_upper_bound:
            rank_upper_bound = new_rank
            curr_best_tri = N.copy()
    return (rank_upper_bound, rank_lower_bound, curr_best_tri)