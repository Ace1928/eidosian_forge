def maxCtxContextualSubtable(maxCtx, st, ruleType, chain=''):
    """Calculate usMaxContext based on a contextual feature subtable."""
    if st.Format == 1:
        for ruleset in getattr(st, '%s%sRuleSet' % (chain, ruleType)):
            if ruleset is None:
                continue
            for rule in getattr(ruleset, '%s%sRule' % (chain, ruleType)):
                if rule is None:
                    continue
                maxCtx = maxCtxContextualRule(maxCtx, rule, chain)
    elif st.Format == 2:
        for ruleset in getattr(st, '%s%sClassSet' % (chain, ruleType)):
            if ruleset is None:
                continue
            for rule in getattr(ruleset, '%s%sClassRule' % (chain, ruleType)):
                if rule is None:
                    continue
                maxCtx = maxCtxContextualRule(maxCtx, rule, chain)
    elif st.Format == 3:
        maxCtx = maxCtxContextualRule(maxCtx, st, chain)
    return maxCtx