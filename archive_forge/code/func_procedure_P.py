from collections import defaultdict
import networkx as nx
def procedure_P(V_minus, V_plus, N, H, F, C, L, excluded_colors=None):
    """Procedure P as described in the paper."""
    if excluded_colors is None:
        excluded_colors = set()
    A_cal = set()
    T_cal = {}
    R_cal = []
    reachable = [V_minus]
    marked = set(reachable)
    idx = 0
    while idx < len(reachable):
        pop = reachable[idx]
        idx += 1
        A_cal.add(pop)
        R_cal.append(pop)
        next_layer = []
        for k in C:
            if H[k, pop] > 0 and k not in A_cal and (k not in excluded_colors) and (k not in marked):
                next_layer.append(k)
        for dst in next_layer:
            T_cal[dst] = pop
        marked.update(next_layer)
        reachable.extend(next_layer)
    b = len(C) - len(A_cal)
    if V_plus in A_cal:
        move_witnesses(V_plus, V_minus, N=N, H=H, F=F, C=C, T_cal=T_cal, L=L)
    else:
        A_0 = set()
        A_cal_0 = set()
        num_terminal_sets_found = 0
        made_equitable = False
        for W_1 in R_cal[::-1]:
            for v in C[W_1]:
                X = None
                for U in C:
                    if N[v, U] == 0 and U in A_cal and (U != W_1):
                        X = U
                if X is None:
                    continue
                for U in C:
                    if N[v, U] >= 1 and U not in A_cal:
                        X_prime = U
                        w = v
                        try:
                            y = next((node for node in L[w] if F[node] == X_prime and N[node, W_1] == 1))
                        except StopIteration:
                            pass
                        else:
                            W = W_1
                            change_color(w, W, X, N=N, H=H, F=F, C=C, L=L)
                            move_witnesses(src_color=X, dst_color=V_minus, N=N, H=H, F=F, C=C, T_cal=T_cal, L=L)
                            change_color(y, X_prime, W, N=N, H=H, F=F, C=C, L=L)
                            procedure_P(V_minus=X_prime, V_plus=V_plus, N=N, H=H, C=C, F=F, L=L, excluded_colors=excluded_colors.union(A_cal))
                            made_equitable = True
                            break
                if made_equitable:
                    break
            else:
                A_cal_0.add(W_1)
                A_0.update(C[W_1])
                num_terminal_sets_found += 1
            if num_terminal_sets_found == b:
                B_cal_prime = set()
                T_cal_prime = {}
                reachable = [V_plus]
                marked = set(reachable)
                idx = 0
                while idx < len(reachable):
                    pop = reachable[idx]
                    idx += 1
                    B_cal_prime.add(pop)
                    next_layer = [k for k in C if H[pop, k] > 0 and k not in B_cal_prime and (k not in marked)]
                    for dst in next_layer:
                        T_cal_prime[pop] = dst
                    marked.update(next_layer)
                    reachable.extend(next_layer)
                I_set = set()
                I_covered = set()
                W_covering = {}
                B_prime = [node for k in B_cal_prime for node in C[k]]
                for z in C[V_plus] + B_prime:
                    if z in I_covered or F[z] not in B_cal_prime:
                        continue
                    I_set.add(z)
                    I_covered.add(z)
                    I_covered.update(list(L[z]))
                    for w in L[z]:
                        if F[w] in A_cal_0 and N[z, F[w]] == 1:
                            if w not in W_covering:
                                W_covering[w] = z
                            else:
                                z_1 = W_covering[w]
                                Z = F[z_1]
                                W = F[w]
                                move_witnesses(W, V_minus, N=N, H=H, F=F, C=C, T_cal=T_cal, L=L)
                                move_witnesses(V_plus, Z, N=N, H=H, F=F, C=C, T_cal=T_cal_prime, L=L)
                                change_color(z_1, Z, W, N=N, H=H, F=F, C=C, L=L)
                                W_plus = next((k for k in C if N[w, k] == 0 and k not in A_cal))
                                change_color(w, W, W_plus, N=N, H=H, F=F, C=C, L=L)
                                excluded_colors.update([k for k in C if k != W and k not in B_cal_prime])
                                procedure_P(V_minus=W, V_plus=W_plus, N=N, H=H, C=C, F=F, L=L, excluded_colors=excluded_colors)
                                made_equitable = True
                                break
                    if made_equitable:
                        break
                else:
                    assert False, 'Must find a w which is the solo neighbor of two vertices in B_cal_prime.'
            if made_equitable:
                break