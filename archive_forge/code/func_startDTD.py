def startDTD(self, name, public_id, system_id):
    """Report the start of the DTD declarations, if the document
        has an associated DTD.

        A startEntity event will be reported before declaration events
        from the external DTD subset are reported, and this can be
        used to infer from which subset DTD declarations derive.

        name is the name of the document element type, public_id the
        public identifier of the DTD (or None if none were supplied)
        and system_id the system identfier of the external subset (or
        None if none were supplied)."""