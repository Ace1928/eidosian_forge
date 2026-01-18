from fontTools.encodings.StandardEncoding import StandardEncoding
def proc_bind(self, proc):
    for i in range(len(proc.value)):
        item = proc.value[i]
        if item.type == 'proceduretype':
            self.proc_bind(item)
        elif not item.literal:
            try:
                obj = self.resolve_name(item.value)
            except:
                pass
            else:
                if obj.type == 'operatortype':
                    proc.value[i] = obj