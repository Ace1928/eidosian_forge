def maxCtxSubtable(maxCtx, tag, lookupType, st):
    """Calculate usMaxContext based on a single lookup table (and an existing
    max value).
    """
    if tag == 'GPOS' and lookupType == 1 or (tag == 'GSUB' and lookupType in (1, 2, 3)):
        maxCtx = max(maxCtx, 1)
    elif tag == 'GPOS' and lookupType == 2:
        maxCtx = max(maxCtx, 2)
    elif tag == 'GSUB' and lookupType == 4:
        for ligatures in st.ligatures.values():
            for ligature in ligatures:
                maxCtx = max(maxCtx, ligature.CompCount)
    elif tag == 'GPOS' and lookupType == 7 or (tag == 'GSUB' and lookupType == 5):
        maxCtx = maxCtxContextualSubtable(maxCtx, st, 'Pos' if tag == 'GPOS' else 'Sub')
    elif tag == 'GPOS' and lookupType == 8 or (tag == 'GSUB' and lookupType == 6):
        maxCtx = maxCtxContextualSubtable(maxCtx, st, 'Pos' if tag == 'GPOS' else 'Sub', 'Chain')
    elif tag == 'GPOS' and lookupType == 9 or (tag == 'GSUB' and lookupType == 7):
        maxCtx = maxCtxSubtable(maxCtx, tag, st.ExtensionLookupType, st.ExtSubTable)
    elif tag == 'GSUB' and lookupType == 8:
        maxCtx = maxCtxContextualRule(maxCtx, st, 'Reverse')
    return maxCtx